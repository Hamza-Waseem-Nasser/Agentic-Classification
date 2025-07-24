"""
ITSM Classification System

A multi-agent AI system for classifying Arabic IT service management tickets.

This package provides:
- Agent framework with pluggable configurations
- Arabic text processing capabilities
- Hierarchical classification workflow
- Multi-tenant support for different companies
"""

__version__ = "0.1.0"
__author__ = "ITSM Classification Team"

# Main exports for easy importing
from .config.agent_config import BaseAgentConfig, CompanyConfig
from .agents.base_agent import BaseAgent

__all__ = [
    "BaseAgentConfig",
    "CompanyConfig", 
    "BaseAgent"
]
