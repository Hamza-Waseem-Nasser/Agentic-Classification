"""
SYSTEM CONFIGURATION VALIDATION

This module provides validation for system configuration to ensure all required
components are properly configured before initializing the classification system.
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator, Field
from pathlib import Path


class SystemConfig(BaseModel):
    """System configuration with validation"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key for LLM operations")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    openai_max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens per request")
    openai_organization_id: Optional[str] = Field(None, description="OpenAI organization ID")
    
    # Qdrant Configuration
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_collection: str = Field(default="itsm_categories", description="Qdrant collection name")
    
    # Data Configuration
    csv_file_path: str = Field(..., description="Path to the category CSV file")
    
    # Processing Configuration
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model for vector search")
    
    # Add strict matching configuration
    strict_category_matching: bool = Field(default=True, description="Enforce exact category name matching")
    allow_fuzzy_matching: bool = Field(default=False, description="Allow fuzzy category matching")
    remove_system_tags: bool = Field(default=True, description="Remove system-generated tags like AutoClosed")
    
    # Add category-specific confidence thresholds
    category_confidence_thresholds: Dict[str, float] = Field(
        default={
            "التسجيل": 0.75,
            "تسجيل الدخول": 0.7,
            "المدفوعات": 0.8,  # Higher for financial
            "الإرسالية": 0.75,
            "إضافة المنتجات": 0.7,
            "default": 0.65
        },
        description="Category-specific confidence thresholds"
    )
    
    # Add common misclassification mappings for logging
    common_misclassifications: Dict[str, str] = Field(
        default={
            "تسجيل": "التسجيل",  # Missing 'ال'
            "دخول": "تسجيل الدخول",  # Partial match
            "المدفوع": "المدفوعات",  # Typo variant
        },
        description="Common misclassification patterns to watch for"
    )
    
    # List of system tags to remove
    system_tags_to_remove: List[str] = Field(
        default=[
            "(AutoClosed)",
            "(auto-closed)", 
            "[closed]",
            "(Closed)",
            "تم الإغلاق تلقائيا",
            "مغلق تلقائيا"
        ],
        description="System tags to remove during processing"
    )
    
    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or v in ["your-api-key-here", "sk-placeholder"]:
            raise ValueError("Valid OpenAI API key required")
        if not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @validator('csv_file_path')
    def validate_csv_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"CSV file not found: {v}")
        if not v.endswith('.csv'):
            raise ValueError("File must be a CSV file")
        return v
    
    @validator('qdrant_url')
    def validate_qdrant_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v
    
    @classmethod
    def from_env(cls, csv_file_path: str = None) -> 'SystemConfig':
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            openai_model=os.getenv('OPENAI_MODEL', 'gpt-4'),
            openai_temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
            openai_max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
            openai_organization_id=os.getenv('OPENAI_ORGANIZATION_ID'),
            qdrant_url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
            qdrant_collection=os.getenv('QDRANT_COLLECTION', 'itsm_categories'),
            csv_file_path=csv_file_path or os.getenv('CSV_FILE_PATH', 'Category + SubCategory.csv'),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            strict_category_matching=os.getenv('STRICT_CATEGORY_MATCHING', 'true').lower() == 'true',
            allow_fuzzy_matching=os.getenv('ALLOW_FUZZY_MATCHING', 'false').lower() == 'true',
            remove_system_tags=os.getenv('REMOVE_SYSTEM_TAGS', 'true').lower() == 'true'
        )
    
    def to_agent_config_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for agent configuration"""
        return {
            'openai': {
                'api_key': self.openai_api_key,
                'model': self.openai_model,
                'temperature': self.openai_temperature,
                'max_tokens': self.openai_max_tokens,
                'organization_id': self.openai_organization_id
            },
            'qdrant': {
                'url': self.qdrant_url,
                'collection_name': self.qdrant_collection
            },
            'classification': {
                'csv_file_path': self.csv_file_path,
                'confidence_threshold': self.confidence_threshold,
                'embedding_model': self.embedding_model,
                'strict_category_matching': self.strict_category_matching,
                'allow_fuzzy_matching': self.allow_fuzzy_matching,
                'remove_system_tags': self.remove_system_tags,
                'system_tags_to_remove': self.system_tags_to_remove
            }
        }


class ConfigValidator:
    """Validates system configuration and dependencies"""
    
    @staticmethod
    def validate_config(config: SystemConfig) -> Dict[str, Any]:
        """
        Validate system configuration and return validation results.
        
        Args:
            config: System configuration to validate
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "dependencies_checked": {}
        }
        
        # Check OpenAI connectivity
        try:
            import openai
            openai.api_key = config.openai_api_key
            validation_result["dependencies_checked"]["openai"] = "available"
        except ImportError:
            validation_result["errors"].append("OpenAI library not installed")
            validation_result["is_valid"] = False
        except Exception as e:
            validation_result["warnings"].append(f"OpenAI setup issue: {str(e)}")
        
        # Check Qdrant connectivity
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=config.qdrant_url)
            # Try to connect
            collections = client.get_collections()
            validation_result["dependencies_checked"]["qdrant"] = "connected"
        except ImportError:
            validation_result["errors"].append("Qdrant client library not installed")
            validation_result["is_valid"] = False
        except Exception as e:
            validation_result["warnings"].append(f"Qdrant connection issue: {str(e)}")
            validation_result["dependencies_checked"]["qdrant"] = "unreachable"
        
        # Check CSV file
        try:
            import pandas as pd
            df = pd.read_csv(config.csv_file_path, encoding='utf-8')
            required_columns = ['Category', 'Category_Description', 'SubCategory', 'SubCategory_Description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_result["errors"].append(f"CSV missing columns: {missing_columns}")
                validation_result["is_valid"] = False
            else:
                validation_result["dependencies_checked"]["csv_structure"] = "valid"
                validation_result["dependencies_checked"]["csv_rows"] = len(df)
                
        except FileNotFoundError:
            validation_result["errors"].append(f"CSV file not found: {config.csv_file_path}")
            validation_result["is_valid"] = False
        except Exception as e:
            validation_result["errors"].append(f"CSV validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    @staticmethod
    def check_system_health(config: SystemConfig) -> Dict[str, Any]:
        """
        Check the health of all system components.
        
        Args:
            config: System configuration
            
        Returns:
            Dict with health check results
        """
        health_status = {
            "qdrant": False,
            "openai": False,
            "csv_loaded": False,
            "overall_healthy": False,
            "checks_performed_at": None,
            "details": {}
        }
        
        from datetime import datetime
        health_status["checks_performed_at"] = datetime.now().isoformat()
        
        # Check Qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=config.qdrant_url)
            collections = client.get_collections()
            health_status["qdrant"] = True
            health_status["details"]["qdrant"] = f"Connected, {len(collections.collections)} collections"
        except Exception as e:
            health_status["details"]["qdrant"] = f"Failed: {str(e)}"
        
        # Check OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.openai_api_key)
            models = client.models.list()
            health_status["openai"] = True
            health_status["details"]["openai"] = "API accessible"
        except Exception as e:
            health_status["details"]["openai"] = f"Failed: {str(e)}"
        
        # Check CSV
        try:
            from ..data.category_loader import CategoryLoader
            loader = CategoryLoader()
            validation = loader.validate_csv_file(config.csv_file_path)
            health_status["csv_loaded"] = validation["is_valid"]
            health_status["details"]["csv"] = f"Valid: {validation['is_valid']}, Rows: {validation.get('row_count', 0)}"
        except Exception as e:
            health_status["details"]["csv"] = f"Failed: {str(e)}"
        
        # Overall health
        health_status["overall_healthy"] = all([
            health_status["qdrant"],
            health_status["openai"],
            health_status["csv_loaded"]
        ])
        
        return health_status
