"""
MAIN INITIALIZATION SCRIPT

This script provides proper initialization of the ITSM classification system
with all the fixes for the identified issues:

1. Async initialization of Qdrant vector collections
2. Proper OpenAI API key validation
3. CSV hierarchy loading
4. Error handling and fallbacks
5. Health checks
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.classification_pipeline import ClassificationPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_classification_system(csv_file_path: str = None, 
                                         config_path: str = None) -> 'ClassificationPipeline':
    """
    Initialize the complete classification system with proper async handling.
    
    Args:
        csv_file_path: Path to the category CSV file
        config_path: Path to configuration file (optional)
        
    Returns:
        Fully initialized ClassificationPipeline
    """
    from src.config.config_validator import SystemConfig, ConfigValidator
    from src.data.category_loader import CategoryLoader
    from src.agents.classification_pipeline import ClassificationPipeline
    from qdrant_client import QdrantClient
    
    logger.info("Starting ITSM Classification System initialization...")
    
    # 1. Load and validate configuration
    logger.info("Loading configuration...")
    try:
        if config_path and os.path.exists(config_path):
            # Load from file if provided
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            config = SystemConfig(**config_data)
        else:
            # Load from environment variables and parameters
            config = SystemConfig.from_env(csv_file_path or "Category + SubCategory.csv")
        
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        raise
    
    # 2. Validate configuration
    logger.info("Validating system configuration...")
    validation_result = ConfigValidator.validate_config(config)
    
    if not validation_result["is_valid"]:
        logger.error("Configuration validation failed:")
        for error in validation_result["errors"]:
            logger.error(f"  - {error}")
        raise ValueError("System configuration is invalid")
    
    if validation_result["warnings"]:
        logger.warning("Configuration warnings:")
        for warning in validation_result["warnings"]:
            logger.warning(f"  - {warning}")
    
    # 3. Initialize Qdrant client
    logger.info("Initializing Qdrant vector database...")
    try:
        qdrant_client = QdrantClient(url=config.qdrant_url)
        # Test connection
        collections = qdrant_client.get_collections()
        logger.info(f"Qdrant connected successfully. Found {len(collections.collections)} collections.")
    except Exception as e:
        logger.warning(f"Qdrant connection failed: {e}")
        logger.info("Continuing without Qdrant (will use keyword fallback)")
        qdrant_client = None
    
    # 4. Load hierarchy from CSV
    logger.info("Loading classification hierarchy from CSV...")
    try:
        loader = CategoryLoader()
        hierarchy = loader.load_from_csv(config.csv_file_path)
        
        stats = loader.get_loading_stats()
        logger.info(f"Hierarchy loaded: {stats.categories_created} categories, "
                   f"{stats.subcategories_created} subcategories")
        
        if stats.warnings:
            logger.warning(f"CSV loading had {len(stats.warnings)} warnings")
            for warning in stats.warnings[:5]:  # Show first 5 warnings
                logger.warning(f"  - {warning}")
                
    except Exception as e:
        logger.error(f"Failed to load hierarchy from CSV: {e}")
        raise
    
    # 5. Create classification pipeline with async initialization
    logger.info("Initializing classification pipeline...")
    try:
        pipeline = await ClassificationPipeline.create(
            config_path=None,  # We already have loaded config
            hierarchy=hierarchy,
            qdrant_client=qdrant_client
        )
        
        # Override with our validated configuration
        pipeline.config = config.to_agent_config_dict()
        
        logger.info("Classification pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        raise
    
    # 6. Perform health check
    logger.info("Performing system health check...")
    health_status = ConfigValidator.check_system_health(config)
    
    if health_status["overall_healthy"]:
        logger.info("✅ System is healthy and ready for classification")
    else:
        logger.warning("⚠️ System has some issues but may still function:")
        for component, status in health_status["details"].items():
            if health_status.get(component, False):
                logger.info(f"  ✅ {component}: {status}")
            else:
                logger.warning(f"  ❌ {component}: {status}")
    
    logger.info("ITSM Classification System initialization completed")
    return pipeline


async def classify_ticket(pipeline, ticket_text: str, ticket_id: str = None) -> dict:
    """
    Classify a single ticket using the initialized pipeline.
    
    Args:
        pipeline: Initialized classification pipeline
        ticket_text: Text content of the ticket
        ticket_id: Optional ticket ID
        
    Returns:
        Classification results
    """
    try:
        logger.info(f"Classifying ticket: {ticket_id or 'auto-generated'}")
        
        # Use the pipeline's classify_ticket method directly
        result = await pipeline.classify_ticket(ticket_text, ticket_id)
        
        logger.info(f"Classification completed for ticket {ticket_id}")
        return result
        
    except Exception as e:
        logger.error(f"Ticket classification failed: {e}")
        return {
            "ticket_id": ticket_id,
            "error": str(e),
            "status": "failed"
        }


def create_sample_config_file(file_path: str = "config.json"):
    """Create a sample configuration file"""
    sample_config = {
        "openai_api_key": "sk-your-api-key-here",
        "openai_model": "gpt-4",
        "openai_temperature": 0.1,
        "openai_max_tokens": 1000,
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "itsm_categories",
        "csv_file_path": "Category + SubCategory.csv",
        "confidence_threshold": 0.7,
        "embedding_model": "text-embedding-3-small"
    }
    
    import json
    with open(file_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample configuration created at: {file_path}")
    print("Please update the OpenAI API key and other settings as needed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ITSM Classification System")
    parser.add_argument("--csv", default="Category + SubCategory.csv", 
                       help="Path to category CSV file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", 
                       help="Create sample configuration file")
    parser.add_argument("--test-ticket", help="Test classification with this text")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config_file()
        exit(0)
    
    async def main():
        try:
            # Initialize system
            pipeline = await initialize_classification_system(args.csv, args.config)
            
            # Test with sample ticket if provided
            if args.test_ticket:
                result = await classify_ticket(pipeline, args.test_ticket)
                print("Classification Result:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("System initialized successfully. Use --test-ticket to test classification.")
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            exit(1)
    
    # Run the async main function
    asyncio.run(main())
