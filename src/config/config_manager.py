"""
Configuration Manager Module

This module provides centralized configuration management for the entire ITSM 
classification system. It handles loading, validation, and distribution of 
configuration to all agents and components.

Key Features:
- Single source of truth for all configuration
- Environment variable integration
- Configuration validation and defaults
- Agent-specific configuration generation
- JSON/file-based configuration support

Usage:
    # Initialize with default config
    config_mgr = ConfigurationManager()
    
    # Initialize with config file
    config_mgr = ConfigurationManager("config.json")
    
    # Get agent configurations
    category_config = config_mgr.get_agent_config("category_classifier")
    subcategory_config = config_mgr.get_agent_config("subcategory_classifier")
    
    # Get LLM factory configuration
    llm_config = config_mgr.get_llm_config()
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict

from .agent_config import BaseAgentConfig, create_default_agent_config
from .config_validator import SystemConfig

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Centralized configuration manager for the ITSM classification system.
    
    This class manages all configuration loading, validation, and distribution
    across agents and components. It ensures consistency and provides a single
    point of configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.raw_config = self._load_raw_config()
        self.system_config = self._create_system_config()
        self.agent_configs = {}
        self._initialize_agent_configs()
    
    def _load_raw_config(self) -> Dict[str, Any]:
        """
        Load raw configuration from file or environment.
        
        Returns:
            Dictionary containing raw configuration data
        """
        try:
            # Try to load from file first
            if self.config_path and os.path.exists(self.config_path):
                logger.info(f"Loading configuration from file: {self.config_path}")
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info("Configuration loaded successfully from file")
                return config
            
            # Fall back to environment variables and defaults
            logger.info("Loading configuration from environment variables")
            return self._load_from_environment()
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Dictionary containing configuration from environment
        """
        return {
            # OpenAI Configuration
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
                "temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "1000")),
                "organization_id": os.getenv("OPENAI_ORG_ID"),
                "timeout": int(os.getenv("OPENAI_TIMEOUT", "30")),
                "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3"))
            },
            
            # Qdrant Configuration
            "qdrant": {
                "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "collection_name": os.getenv("QDRANT_COLLECTION", "itsm_categories"),
                "timeout": int(os.getenv("QDRANT_TIMEOUT", "30"))
            },
            
            # Data Configuration
            "data": {
                "csv_file_path": os.getenv("CSV_FILE_PATH", "Category + SubCategory.csv"),
                "hierarchy_path": os.getenv("HIERARCHY_PATH", "data/classification_hierarchy.json")
            },
            
            # Classification Configuration
            "classification": {
                "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                "strict_mode": os.getenv("STRICT_MODE", "true").lower() == "true",
                "remove_system_tags": os.getenv("REMOVE_SYSTEM_TAGS", "true").lower() == "true"
            },
            
            # Agent-specific Configuration
            "agents": {
                "orchestrator": {
                    "timeout_seconds": int(os.getenv("ORCHESTRATOR_TIMEOUT", "30"))
                },
                "arabic_processor": {
                    "timeout_seconds": int(os.getenv("ARABIC_PROCESSOR_TIMEOUT", "30"))
                },
                "category_classifier": {
                    "timeout_seconds": int(os.getenv("CATEGORY_CLASSIFIER_TIMEOUT", "45")),
                    "top_k_candidates": int(os.getenv("CATEGORY_TOP_K", "5")),
                    "confidence_threshold": float(os.getenv("CATEGORY_CONFIDENCE", "0.75"))
                },
                "subcategory_classifier": {
                    "timeout_seconds": int(os.getenv("SUBCATEGORY_CLASSIFIER_TIMEOUT", "45")),
                    "top_k_subcategories": int(os.getenv("SUBCATEGORY_TOP_K", "3")),
                    "confidence_threshold": float(os.getenv("SUBCATEGORY_CONFIDENCE", "0.6"))
                }
            }
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration when all else fails.
        
        Returns:
            Dictionary containing minimal default configuration
        """
        return {
            "openai": {
                "api_key": None,  # Must be provided
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000,
                "organization_id": None,
                "timeout": 30,
                "max_retries": 3
            },
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "itsm_categories",
                "timeout": 30
            },
            "data": {
                "csv_file_path": "Category + SubCategory.csv",
                "hierarchy_path": "data/classification_hierarchy.json"
            },
            "classification": {
                "confidence_threshold": 0.7,
                "embedding_model": "text-embedding-3-small",
                "strict_mode": True,
                "remove_system_tags": True
            },
            "agents": {
                "orchestrator": {"timeout_seconds": 30},
                "arabic_processor": {"timeout_seconds": 30},
                "category_classifier": {
                    "timeout_seconds": 45,
                    "top_k_candidates": 5,
                    "confidence_threshold": 0.75
                },
                "subcategory_classifier": {
                    "timeout_seconds": 45,
                    "top_k_subcategories": 3,
                    "confidence_threshold": 0.6
                }
            }
        }
    
    def _create_system_config(self) -> SystemConfig:
        """
        Create SystemConfig instance from raw configuration.
        
        Returns:
            Validated SystemConfig instance
        """
        try:
            # Extract system-level configuration
            system_config_data = {
                "openai_api_key": self.raw_config["openai"]["api_key"],
                "openai_model": self.raw_config["openai"]["model"],
                "openai_temperature": self.raw_config["openai"]["temperature"],
                "openai_max_tokens": self.raw_config["openai"]["max_tokens"],
                "openai_organization_id": self.raw_config["openai"].get("organization_id"),
                "qdrant_url": self.raw_config["qdrant"]["url"],
                "qdrant_collection": self.raw_config["qdrant"]["collection_name"],
                "csv_file_path": self.raw_config["data"]["csv_file_path"],
                "confidence_threshold": self.raw_config["classification"]["confidence_threshold"],
                "embedding_model": self.raw_config["classification"]["embedding_model"],
                "strict_category_matching": self.raw_config["classification"]["strict_mode"],
                "remove_system_tags": self.raw_config["classification"]["remove_system_tags"]
            }
            
            return SystemConfig(**system_config_data)
            
        except Exception as e:
            logger.error(f"Failed to create SystemConfig: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def _initialize_agent_configs(self):
        """
        Initialize all agent configurations.
        """
        # Get common OpenAI settings
        openai_config = self.raw_config["openai"]
        
        # Create configuration for each agent
        agent_names = ["orchestrator", "arabic_processor", "category_classifier", "subcategory_classifier"]
        
        for agent_name in agent_names:
            try:
                # Get agent-specific settings
                agent_settings = self.raw_config["agents"].get(agent_name, {})
                
                # Create BaseAgentConfig with merged settings
                config_data = {
                    "agent_name": agent_name,
                    "model_name": openai_config["model"],
                    "temperature": openai_config["temperature"],
                    "max_tokens": openai_config["max_tokens"],
                    "api_key": openai_config["api_key"],
                    "organization_id": openai_config.get("organization_id"),
                    "timeout_seconds": agent_settings.get("timeout_seconds", openai_config["timeout"]),
                    "retry_attempts": openai_config["max_retries"],
                    "confidence_threshold": agent_settings.get("confidence_threshold", 
                                                              self.raw_config["classification"]["confidence_threshold"])
                }
                
                self.agent_configs[agent_name] = BaseAgentConfig(**config_data)
                logger.debug(f"Created configuration for agent: {agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to create config for agent {agent_name}: {e}")
                # Create basic config as fallback
                self.agent_configs[agent_name] = create_default_agent_config(agent_name)
    
    def get_agent_config(self, agent_name: str) -> BaseAgentConfig:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            BaseAgentConfig for the specified agent
            
        Raises:
            ValueError: If agent_name is not recognized
        """
        if agent_name not in self.agent_configs:
            logger.warning(f"Unknown agent name: {agent_name}, creating default config")
            return create_default_agent_config(agent_name)
        
        return self.agent_configs[agent_name]
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for the factory.
        
        Returns:
            Dictionary containing LLM configuration
        """
        return {
            "api_key": self.raw_config["openai"]["api_key"],
            "model": self.raw_config["openai"]["model"],
            "temperature": self.raw_config["openai"]["temperature"],
            "max_tokens": self.raw_config["openai"]["max_tokens"],
            "organization_id": self.raw_config["openai"].get("organization_id"),
            "timeout": self.raw_config["openai"]["timeout"],
            "max_retries": self.raw_config["openai"]["max_retries"]
        }
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """
        Get Qdrant configuration.
        
        Returns:
            Dictionary containing Qdrant configuration
        """
        return self.raw_config["qdrant"]
    
    def get_classification_config(self) -> Dict[str, Any]:
        """
        Get classification-specific configuration.
        
        Returns:
            Dictionary containing classification configuration
        """
        return self.raw_config["classification"]
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Dictionary containing data configuration
        """
        return self.raw_config["data"]
    
    def get_system_config(self) -> SystemConfig:
        """
        Get the system configuration object.
        
        Returns:
            Validated SystemConfig instance
        """
        return self.system_config
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the entire configuration.
        
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "agent_validations": {}
        }
        
        # Validate OpenAI API key
        if not self.raw_config["openai"]["api_key"]:
            validation_result["is_valid"] = False
            validation_result["errors"].append("OpenAI API key is required")
        
        # Validate agent configurations
        from .llm_factory import LLMFactory
        
        for agent_name, config in self.agent_configs.items():
            agent_validation = LLMFactory.validate_config(config)
            validation_result["agent_validations"][agent_name] = agent_validation
            
            if not agent_validation["is_valid"]:
                validation_result["is_valid"] = False
                for error in agent_validation["errors"]:
                    validation_result["errors"].append(f"{agent_name}: {error}")
            
            validation_result["warnings"].extend([
                f"{agent_name}: {warning}" for warning in agent_validation["warnings"]
            ])
        
        # Validate file paths
        csv_path = self.raw_config["data"]["csv_file_path"]
        if csv_path and not os.path.exists(csv_path):
            validation_result["warnings"].append(f"CSV file not found: {csv_path}")
        
        return validation_result
    
    def update_agent_config(self, agent_name: str, **kwargs):
        """
        Update configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent to update
            **kwargs: Configuration parameters to update
        """
        if agent_name in self.agent_configs:
            # Create new config with updated values
            current_config = self.agent_configs[agent_name]
            config_dict = asdict(current_config)
            config_dict.update(kwargs)
            
            self.agent_configs[agent_name] = BaseAgentConfig(**config_dict)
            logger.info(f"Updated configuration for agent: {agent_name}")
        else:
            logger.warning(f"Cannot update unknown agent: {agent_name}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Dictionary containing health check results
        """
        health_status = {
            "overall_healthy": True,
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Check OpenAI API key
        api_key = self.raw_config["openai"]["api_key"]
        if not api_key or api_key in ["your-api-key-here", "sk-your-api-key-here"]:
            health_status["overall_healthy"] = False
            health_status["details"]["openai_api_key"] = "❌ Invalid or missing API key"
            health_status["errors"].append("OpenAI API key is required")
        else:
            health_status["details"]["openai_api_key"] = "✅ API key configured"
        
        # Check CSV file
        csv_path = self.raw_config["data"]["csv_file_path"]
        if os.path.exists(csv_path):
            health_status["details"]["csv_file"] = f"✅ CSV file found: {csv_path}"
        else:
            health_status["overall_healthy"] = False
            health_status["details"]["csv_file"] = f"❌ CSV file not found: {csv_path}"
            health_status["errors"].append(f"CSV file not found: {csv_path}")
        
        # Check Qdrant connection
        try:
            from qdrant_client import QdrantClient
            qdrant_config = self.get_qdrant_config()
            client = QdrantClient(url=qdrant_config["url"])
            collections = client.get_collections()
            health_status["details"]["qdrant"] = f"✅ Qdrant accessible with {len(collections.collections)} collections"
        except Exception as e:
            health_status["details"]["qdrant"] = f"⚠️ Qdrant connection failed: {str(e)[:50]}..."
            health_status["warnings"].append(f"Qdrant connection issue: {e}")
        
        # Check agent configurations
        for agent_name in self.agent_configs:
            try:
                config = self.get_agent_config(agent_name)
                if config.api_key:
                    health_status["details"][f"agent_{agent_name}"] = "✅ Agent configured"
                else:
                    health_status["details"][f"agent_{agent_name}"] = "⚠️ Agent missing API key"
                    health_status["warnings"].append(f"Agent {agent_name} missing API key")
            except Exception as e:
                health_status["overall_healthy"] = False
                health_status["details"][f"agent_{agent_name}"] = f"❌ Agent config error: {e}"
                health_status["errors"].append(f"Agent {agent_name} configuration error: {e}")
        
        return health_status

    def save_config(self, output_path: str):
        """
        Save current configuration to a file.
        
        Args:
            output_path: Path where to save the configuration
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.raw_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_agent_names(self) -> List[str]:
        """
        Get list of all configured agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agent_configs.keys())
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"ConfigurationManager(agents={len(self.agent_configs)}, config_path={self.config_path})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration manager."""
        return (f"ConfigurationManager("
                f"config_path='{self.config_path}', "
                f"agents={list(self.agent_configs.keys())}, "
                f"openai_model='{self.raw_config['openai']['model']}'"
                f")")


def create_configuration_manager(config_path: Optional[str] = None) -> ConfigurationManager:
    """
    Factory function to create a configuration manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured ConfigurationManager instance
    """
    return ConfigurationManager(config_path)


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Utility function to load configuration from a JSON file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_sample_config_file(output_path: str = "itsm_config.json"):
    """
    Create a sample configuration file with all available options.
    
    Args:
        output_path: Where to save the sample configuration
    """
    sample_config = {
        "openai": {
            "api_key": "sk-your-openai-api-key-here",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 1000,
            "organization_id": None,
            "timeout": 30,
            "max_retries": 3
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "collection_name": "itsm_categories",
            "timeout": 30
        },
        "data": {
            "csv_file_path": "Category + SubCategory.csv",
            "hierarchy_path": "data/classification_hierarchy.json"
        },
        "classification": {
            "confidence_threshold": 0.7,
            "embedding_model": "text-embedding-3-small",
            "strict_mode": True,
            "remove_system_tags": True
        },
        "agents": {
            "orchestrator": {
                "timeout_seconds": 30
            },
            "arabic_processor": {
                "timeout_seconds": 30
            },
            "category_classifier": {
                "timeout_seconds": 45,
                "top_k_candidates": 5,
                "confidence_threshold": 0.75
            },
            "subcategory_classifier": {
                "timeout_seconds": 45,
                "top_k_subcategories": 3,
                "confidence_threshold": 0.6
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"Sample configuration created at: {output_path}")
    print("Please update the OpenAI API key and other settings as needed.")
