"""
LLM Factory Module

This module provides a centralized factory for creating LLM instances with consistent 
configuration across the entire ITSM classification system.

Key Benefits:
- Single point of configuration for all LLM instances
- Consistent API key management
- Easy switching between different LLM providers
- Simplified testing with mock LLMs
- Centralized error handling for LLM initialization

Usage:
    # Create ChatOpenAI for LangChain integration
    config = BaseAgentConfig(agent_name="test", api_key="sk-...")
    llm = LLMFactory.create_chat_llm(config)
    
    # Create AsyncOpenAI for embeddings
    openai_client = LLMFactory.create_async_openai(config)
    
    # Create from dictionary config
    llm = LLMFactory.create_from_config_dict({
        "model": "gpt-4",
        "temperature": 0.1,
        "api_key": "sk-..."
    })
"""

from typing import Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from .agent_config import BaseAgentConfig

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory for creating LLM instances with consistent configuration.
    
    This factory ensures all LLM instances across the system are created
    with the same configuration patterns and error handling.
    """
    
    @staticmethod
    def create_chat_llm(config: BaseAgentConfig) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance for LangChain integration.
        
        Args:
            config: BaseAgentConfig containing LLM settings
            
        Returns:
            Configured ChatOpenAI instance
            
        Raises:
            ValueError: If configuration is invalid
            Exception: If LLM initialization fails
        """
        try:
            # Validate required configuration
            if not config.api_key:
                raise ValueError("OpenAI API key is required but not provided in config")
            
            if config.api_key == "your-api-key-here" or config.api_key == "sk-your-api-key-here":
                raise ValueError("Valid OpenAI API key required - please set a real API key")
            
            # Create ChatOpenAI instance with standardized configuration
            llm = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
                organization=config.organization_id,
                timeout=config.timeout_seconds,
                max_retries=config.retry_attempts
            )
            
            logger.info(f"Created ChatOpenAI instance: {config.model_name} for {config.agent_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create ChatOpenAI instance for {config.agent_name}: {e}")
            raise
    
    @staticmethod
    def create_async_openai(config: BaseAgentConfig) -> AsyncOpenAI:
        """
        Create an AsyncOpenAI client for embeddings and direct API calls.
        
        Args:
            config: BaseAgentConfig containing API settings
            
        Returns:
            Configured AsyncOpenAI client
            
        Raises:
            ValueError: If configuration is invalid
            Exception: If client initialization fails
        """
        try:
            # Validate required configuration
            if not config.api_key:
                raise ValueError("OpenAI API key is required for AsyncOpenAI client")
            
            if config.api_key == "your-api-key-here" or config.api_key == "sk-your-api-key-here":
                raise ValueError("Valid OpenAI API key required - please set a real API key")
            
            # Create AsyncOpenAI client
            client = AsyncOpenAI(
                api_key=config.api_key,
                organization=config.organization_id,
                timeout=config.timeout_seconds,
                max_retries=config.retry_attempts
            )
            
            logger.info(f"Created AsyncOpenAI client for {config.agent_name}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create AsyncOpenAI client for {config.agent_name}: {e}")
            raise
    
    @staticmethod
    def create_from_config_dict(config_dict: Dict[str, Any], agent_name: str = "unknown") -> ChatOpenAI:
        """
        Create a ChatOpenAI instance from a dictionary configuration.
        
        This method is useful when you have configuration from JSON files
        or other sources that aren't BaseAgentConfig objects.
        
        Args:
            config_dict: Dictionary containing LLM configuration
            agent_name: Name of the agent using this LLM
            
        Returns:
            Configured ChatOpenAI instance
            
        Raises:
            ValueError: If required keys are missing from config_dict
            Exception: If LLM initialization fails
        """
        try:
            # Extract configuration with defaults
            api_key = config_dict.get('api_key') or config_dict.get('openai_api_key')
            model = config_dict.get('model') or config_dict.get('model_name', 'gpt-4o-mini')
            temperature = config_dict.get('temperature', 0.1)
            max_tokens = config_dict.get('max_tokens', 1000)
            organization_id = config_dict.get('organization_id') or config_dict.get('openai_organization_id')
            timeout = config_dict.get('timeout') or config_dict.get('timeout_seconds', 30)
            max_retries = config_dict.get('max_retries') or config_dict.get('retry_attempts', 3)
            
            # Validate required fields
            if not api_key:
                raise ValueError("API key is required in config_dict (use 'api_key' or 'openai_api_key' key)")
            
            if api_key in ["your-api-key-here", "sk-your-api-key-here"]:
                raise ValueError("Valid OpenAI API key required - please set a real API key")
            
            # Create ChatOpenAI instance
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                organization=organization_id,
                timeout=timeout,
                max_retries=max_retries
            )
            
            logger.info(f"Created ChatOpenAI from dict config: {model} for {agent_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create ChatOpenAI from config dict for {agent_name}: {e}")
            raise
    
    @staticmethod
    def create_async_openai_from_dict(config_dict: Dict[str, Any], agent_name: str = "unknown") -> AsyncOpenAI:
        """
        Create an AsyncOpenAI client from a dictionary configuration.
        
        Args:
            config_dict: Dictionary containing API configuration
            agent_name: Name of the agent using this client
            
        Returns:
            Configured AsyncOpenAI client
            
        Raises:
            ValueError: If required keys are missing from config_dict
            Exception: If client initialization fails
        """
        try:
            # Extract configuration
            api_key = config_dict.get('api_key') or config_dict.get('openai_api_key')
            organization_id = config_dict.get('organization_id') or config_dict.get('openai_organization_id')
            timeout = config_dict.get('timeout') or config_dict.get('timeout_seconds', 30)
            max_retries = config_dict.get('max_retries') or config_dict.get('retry_attempts', 3)
            
            # Validate required fields
            if not api_key:
                raise ValueError("API key is required in config_dict (use 'api_key' or 'openai_api_key' key)")
            
            if api_key in ["your-api-key-here", "sk-your-api-key-here"]:
                raise ValueError("Valid OpenAI API key required - please set a real API key")
            
            # Create AsyncOpenAI client
            client = AsyncOpenAI(
                api_key=api_key,
                organization=organization_id,
                timeout=timeout,
                max_retries=max_retries
            )
            
            logger.info(f"Created AsyncOpenAI client from dict config for {agent_name}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create AsyncOpenAI from config dict for {agent_name}: {e}")
            raise
    
    @staticmethod
    def validate_config(config: BaseAgentConfig) -> Dict[str, Any]:
        """
        Validate LLM configuration without creating an instance.
        
        Args:
            config: BaseAgentConfig to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if not config.api_key:
            validation_result["is_valid"] = False
            validation_result["errors"].append("OpenAI API key is required")
        elif config.api_key in ["your-api-key-here", "sk-your-api-key-here"]:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Valid OpenAI API key required (placeholder detected)")
        
        # Check temperature range
        if not (0.0 <= config.temperature <= 2.0):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Temperature must be between 0.0 and 2.0, got {config.temperature}")
        
        # Check confidence threshold
        if not (0.0 <= config.confidence_threshold <= 1.0):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Confidence threshold must be between 0.0 and 1.0, got {config.confidence_threshold}")
        
        # Check max_tokens
        if config.max_tokens <= 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Max tokens must be positive, got {config.max_tokens}")
        
        # Warnings for unusual values
        if config.temperature > 1.0:
            validation_result["warnings"].append(f"High temperature {config.temperature} may produce unpredictable results")
        
        if config.max_tokens > 4000:
            validation_result["warnings"].append(f"High max_tokens {config.max_tokens} may be expensive")
        
        return validation_result


class LLMConfigBuilder:
    """
    Builder pattern for creating LLM configurations.
    
    This provides a fluent interface for building complex LLM configurations
    with validation and default value handling.
    """
    
    def __init__(self, agent_name: str):
        self.config_data = {
            "agent_name": agent_name,
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 1000,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "confidence_threshold": 0.7
        }
    
    def with_model(self, model_name: str) -> 'LLMConfigBuilder':
        """Set the model name"""
        self.config_data["model_name"] = model_name
        return self
    
    def with_temperature(self, temperature: float) -> 'LLMConfigBuilder':
        """Set the temperature"""
        self.config_data["temperature"] = temperature
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'LLMConfigBuilder':
        """Set max tokens"""
        self.config_data["max_tokens"] = max_tokens
        return self
    
    def with_api_key(self, api_key: str) -> 'LLMConfigBuilder':
        """Set API key"""
        self.config_data["api_key"] = api_key
        return self
    
    def with_timeout(self, timeout_seconds: int) -> 'LLMConfigBuilder':
        """Set timeout"""
        self.config_data["timeout_seconds"] = timeout_seconds
        return self
    
    def with_retries(self, retry_attempts: int) -> 'LLMConfigBuilder':
        """Set retry attempts"""
        self.config_data["retry_attempts"] = retry_attempts
        return self
    
    def with_confidence_threshold(self, threshold: float) -> 'LLMConfigBuilder':
        """Set confidence threshold"""
        self.config_data["confidence_threshold"] = threshold
        return self
    
    def build(self) -> BaseAgentConfig:
        """Build the final configuration"""
        return BaseAgentConfig(**self.config_data)
