"""
Configuration Module

Provides configuration classes and utilities for the ITSM classification system.
"""

from .agent_config import BaseAgentConfig, CompanyConfig, create_default_agent_config, create_company_config

__all__ = [
    "BaseAgentConfig",
    "CompanyConfig",
    "create_default_agent_config", 
    "create_company_config"
]
