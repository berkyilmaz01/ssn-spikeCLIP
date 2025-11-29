"""
Professional logging utility for SpikeCLIP.
This module provides structured logging without modifying existing print statements.
Import and use this alongside your existing print statements.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "spikeclip",
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup professional logging configuration.
    
    Usage:
        from utils.logger import setup_logger
        logger = setup_logger()
        logger.info("Training started")
        logger.error("Error occurred")
    
    :param name: Logger name
    :param log_dir: Directory for log files (default: logs/)
    :param log_level: Logging level (default: INFO)
    :param log_to_file: Whether to log to file
    :param log_to_console: Whether to log to console
    :param format_string: Custom log format string
    :return: Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"spikeclip_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "spikeclip") -> logging.Logger:
    """
    Get logger instance. Creates default logger if not configured.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger()
        logger.info("Message")
    
    :param name: Logger name
    :return: Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Setup default logging if not configured
        setup_logger(name=name)
    return logger

