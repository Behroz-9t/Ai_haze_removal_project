"""Logging utility.

Provides a clean, readable console logger with distinct formatting for
INFO, WARNING, and ERROR levels without heavy third-party dependencies.
"""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a pre-configured logger with readable console output.

    Args:
        name: Name for the logger instance. Defaults to root logger if None.
        level: Logging severity level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name if name else "dehazing_system")
    
    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs in nested loggers
        logger.propagate = False

    return logger
