#!/usr/bin/env python3
"""
Centralized logging configuration for all training and inference scripts.
Provides consistent logging to both console and log files.
"""

import sys
import logging
from pathlib import Path


class DualLogger:
    """Log messages to both console (stdout) and file simultaneously"""
    
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.terminal = sys.stdout
        
    def write(self, message):
        """Write message to both terminal and file"""
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        """Flush both streams"""
        self.terminal.flush()
        self.log_file.flush()
        
    def isatty(self):
        """Check if terminal"""
        return self.terminal.isatty()
        
    def close(self):
        """Close the log file"""
        self.log_file.close()


def setup_logging(stream_name, script_name, output_dir=None):
    """
    Setup dual logging (console + file) for training/inference scripts.
    
    Args:
        stream_name: Stream identifier (e.g., 'stream_a_text', 'stream_b_url', 'stream_c_attachments')
        script_name: Name of the script (e.g., 'train', 'evaluate', 'test')
        output_dir: Optional custom output directory. If None, uses training_logs/stream_name
    
    Returns:
        tuple: (DualLogger instance, log_file_path)
    
    Example:
        logger, log_file = setup_logging('stream_a_text', 'train_distilbert')
        sys.stdout = logger
    """
    
    # Determine log directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / 'training_logs' / stream_name
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file (same name, overwrites previous)
    log_filename = f"{script_name}.log"
    log_file_path = output_dir / log_filename
    
    # Initialize dual logger
    logger = DualLogger(str(log_file_path))
    
    return logger, log_file_path


def restore_stdout(logger):
    """Safely restore stdout and close logger"""
    if hasattr(logger, 'close'):
        logger.close()
    sys.stdout = sys.__stdout__
