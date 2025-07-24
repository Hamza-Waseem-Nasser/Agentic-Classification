"""
Agent Configuration Module

This module defines the configuration classes for all agents in the ITSM system.
It provides a flexible, type-safe way to configure agent behavior and LLM settings.

Educational Notes:
- @dataclass automatically generates __init__, __repr__, __eq__ methods
- Type hints make the code self-documenting and enable IDE support
- Default values make configs easy to use while allowing customization
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class BaseAgentConfig:
    """
    Base configuration for all agents in the system.
    
    This class defines the common settings that every agent needs:
    - agent_name: Unique identifier for the agent
    - model_name: Which LLM model to use (gpt-4o, gpt-4-turbo, etc.)
    - temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
    - max_tokens: Maximum response length
    - timeout_seconds: How long to wait for LLM response
    - retry_attempts: How many times to retry on failure
    - confidence_threshold: Minimum confidence score to accept results
    
    Educational: @dataclass automatically creates:
    - __init__(self, agent_name, model_name="gpt-4o", ...)
    - __repr__(self) for nice string representation
    - __eq__(self, other) for comparing configs
    """
    
    # Required field - must be provided when creating config
    agent_name: str
    
    # Optional fields with smart defaults from environment or fallback values
    model_name: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o"))
    temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "1000")))
    
    # Agent behavior settings
    timeout_seconds: int = 30
    retry_attempts: int = 3
    confidence_threshold: float = 0.7
    
    # API Configuration
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    organization_id: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_ORG_ID"))
    
    def __post_init__(self):
        """
        Called automatically after __init__ to validate the configuration.
        
        Educational: This is a special dataclass method that runs after
        the automatic __init__ method. Perfect for validation!
        """
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide api_key in config."
            )
        
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")


@dataclass 
class CompanyConfig:
    """
    Company-specific configuration for multi-tenant support.
    
    Each company can have different:
    - Arabic dialects and technical terms
    - Category hierarchies and confidence thresholds
    - Business rules and escalation procedures
    
    Educational: This shows how to use dataclass with complex default values
    using field(default_factory=...) for mutable defaults like dictionaries.
    """
    
    company_id: str
    company_name: str
    
    # Arabic Processing Configuration
    arabic_dialect: str = "MSA"  # MSA, Gulf, Levantine, Egyptian
    technical_glossary: Dict[str, str] = field(default_factory=dict)
    
    # Classification Configuration  
    category_hierarchy: Dict[str, list] = field(default_factory=dict)
    custom_categories: list = field(default_factory=list)
    
    # Confidence thresholds for different classification levels
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "main_category": 0.8,
        "subcategory": 0.7, 
        "human_escalation": 0.5
    })
    
    # Business Rules
    priority_keywords: list = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)


def create_default_agent_config(agent_name: str, **overrides) -> BaseAgentConfig:
    """
    Factory function to create agent configurations with custom overrides.
    
    Args:
        agent_name: Name of the agent
        **overrides: Any config values to override (model_name, temperature, etc.)
    
    Returns:
        BaseAgentConfig with specified overrides
        
    Example:
        config = create_default_agent_config(
            "arabic_processor",
            temperature=0.0,  # More deterministic for text processing
            max_tokens=500
        )
    """
    # Create base config
    config_data = {"agent_name": agent_name}
    
    # Apply any overrides
    config_data.update(overrides)
    
    return BaseAgentConfig(**config_data)


def create_company_config(company_id: str, company_name: str, **overrides) -> CompanyConfig:
    """
    Factory function to create company configurations.
    
    Args:
        company_id: Unique identifier for the company
        company_name: Human-readable company name
        **overrides: Any company-specific settings
        
    Returns:
        CompanyConfig with specified settings
    """
    config_data = {
        "company_id": company_id,
        "company_name": company_name
    }
    
    config_data.update(overrides)
    
    return CompanyConfig(**config_data)


# Educational Example: How to use these configurations
if __name__ == "__main__":
    """
    This section demonstrates how to use the configuration classes.
    Run this file directly to see examples in action!
    """
    
    print("=== ITSM Agent Configuration Examples ===\n")
    
    # Example 1: Basic agent config
    try:
        basic_config = BaseAgentConfig(agent_name="test_agent")
        print(f"✅ Basic Config: {basic_config}")
        print(f"   Model: {basic_config.model_name}")
        print(f"   Temperature: {basic_config.temperature}")
        print()
    except ValueError as e:
        print(f"❌ Config Error: {e}")
        print("   Make sure to set OPENAI_API_KEY in your .env file!\n")
    
    # Example 2: Custom agent config
    custom_config = create_default_agent_config(
        "arabic_processor",
        temperature=0.0,  # Deterministic for text processing
        confidence_threshold=0.9
    )
    print(f"✅ Custom Config: {custom_config}")
    print()
    
    # Example 3: Company config
    company_config = create_company_config(
        company_id="acme_corp",
        company_name="ACME Corporation",
        arabic_dialect="Gulf",
        confidence_thresholds={
            "main_category": 0.85,
            "subcategory": 0.75,
            "human_escalation": 0.6
        }
    )
    print(f"✅ Company Config: {company_config}")
    print(f"   Dialect: {company_config.arabic_dialect}")
    print(f"   Main Category Threshold: {company_config.confidence_thresholds['main_category']}")
