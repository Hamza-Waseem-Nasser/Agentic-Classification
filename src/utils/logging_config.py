"""
Centralized logging configuration to prevent duplicate log messages.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, 
                  format_string: Optional[str] = None,
                  force_reset: bool = False) -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        force_reset: Force reset of existing handlers (default: False)
    """
    # Check if root logger already has handlers (to prevent duplicate setup)
    root_logger = logging.getLogger()
    
    if root_logger.handlers and not force_reset:
        # Logging already configured, just adjust level if needed
        root_logger.setLevel(level)
        return
    
    # Clear any existing handlers if force_reset is True
    if force_reset:
        root_logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=force_reset  # This ensures we don't add duplicate handlers
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add additional handlers - use parent logger's handlers
    # This prevents duplicate messages
    logger.propagate = True
    
    return logger


def disable_duplicate_handlers():
    """
    Remove duplicate handlers from all existing loggers.
    This is a utility function to clean up existing logger configurations.
    """
    # Get all existing loggers
    loggers = [logging.getLogger()]  # Root logger
    loggers += [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
    
    for logger in loggers:
        # Remove duplicate handlers (keep only the first unique handler of each type)
        seen_handlers = set()
        handlers_to_keep = []
        
        for handler in logger.handlers:
            handler_type = type(handler)
            if handler_type not in seen_handlers:
                seen_handlers.add(handler_type)
                handlers_to_keep.append(handler)
        
        # Clear all handlers and add back only unique ones
        logger.handlers.clear()
        logger.handlers.extend(handlers_to_keep)


def prevent_logger_propagation_issues():
    """
    Ensure all agent loggers have proper propagation settings.
    This prevents duplicate messages from agent loggers.
    """
    agent_logger_prefixes = [
        'agent.',
        'src.agents.',
        'agent.orchestrator',
        'agent.arabic_processor', 
        'agent.category_classifier',
        'agent.subcategory_classifier'
    ]
    
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if any(logger_name.startswith(prefix) for prefix in agent_logger_prefixes):
            logger = logging.getLogger(logger_name)
            # Ensure proper propagation (should be True to use parent handlers)
            logger.propagate = True
            # Remove any duplicate handlers from agent loggers
            if logger.handlers:
                logger.handlers.clear()


def debug_logging_setup():
    """
    Debug function to print current logging configuration.
    Useful for troubleshooting duplicate log messages.
    """
    print("=== LOGGING DEBUG INFO ===")
    
    # Root logger info
    root = logging.getLogger()
    print(f"Root logger level: {root.level}")
    print(f"Root logger handlers: {len(root.handlers)}")
    for i, handler in enumerate(root.handlers):
        print(f"  Handler {i}: {type(handler).__name__}")
    
    # Check all loggers
    print(f"\nTotal loggers: {len(logging.Logger.manager.loggerDict)}")
    
    agent_loggers = []
    for name in logging.Logger.manager.loggerDict:
        if 'agent' in name:
            logger = logging.getLogger(name)
            agent_loggers.append((name, logger))
    
    print(f"Agent loggers: {len(agent_loggers)}")
    for name, logger in agent_loggers[:10]:  # Show first 10
        print(f"  {name}: handlers={len(logger.handlers)}, propagate={logger.propagate}")
    
    print("=== END DEBUG INFO ===")
