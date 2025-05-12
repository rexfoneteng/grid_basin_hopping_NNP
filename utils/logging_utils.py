#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-05-03 00:50:53
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   PC_user
# @Last modified time: 2025-05-03 11:15:38
# @Description: logging utils

import logging
import time
from typing import Optional

class FlushingFileHandler(logging.FileHandler):
    """A file handler that can flush after every N records."""
    
    def __init__(self, filename, mode='a', encoding=None, delay=False, flush_every=1):
        """
        Initialize the handler with flush control.
        
        Args:
            filename: Log file path
            mode: File open mode
            encoding: File encoding
            delay: Whether to delay opening the file
            flush_every: Flush after this many log records
        """
        super().__init__(filename, mode, encoding, delay)
        self.flush_every = flush_every
        self.record_count = 0
    
    def emit(self, record):
        """Emit a record and flush if needed."""
        super().emit(record)
        self.record_count += 1
        
        # Always flush immediately for ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            self.flush()
            return
            
        # Flush based on count
        if self.record_count >= self.flush_every:
            self.flush()
            self.record_count = 0

def setup_logger(
    name: str = "sugar_sampling",
    level: str = "INFO",
    log_file: Optional[str] = "structure_sampling.log",
    flush_every: int = 100,
    max_time_between_flush: Optional[float] = None
) -> logging.Logger:
    """Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, or None for console-only logging
        flush_every: Flush logs to disk after this many records
        max_time_between_flush: Maximum time (in seconds) between flushes
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, level, logging.INFO)
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = FlushingFileHandler(
            log_file, 
            mode="a", 
            encoding="utf-8",
            flush_every=flush_every
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Add time-based flushing if specified
        if max_time_between_flush:
            # Define and start a timer for periodic flushing
            def flush_timer():
                while True:
                    time.sleep(max_time_between_flush)
                    file_handler.flush()
            
            import threading
            flush_thread = threading.Thread(target=flush_timer, daemon=True)
            flush_thread.start()
    
    return logger