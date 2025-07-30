"""
MAIN INITIALIZATION SCRIPT (UPDATED)

This script provides proper initialization of the ITSM classification system
using the new centralized configuration management and LLM factory pattern.

Key improvements:
1. Uses ConfigurationManager for centralized configuration
2. Uses LLMFactory for consistent LLM instance creation
3. Simplified configuration loading and validation
4. Better error handling and fallbacks
5. Consistent API key management

Usage:
    python main_updated.py --csv "Category + SubCategory.csv" --test-ticket "تسجيل الدخول مشكلة"
"""

import asyncio
import logging
import os
import json
import argparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.classification_pipeline import ClassificationPipeline

# Set up centralized logging
from src.utils.logging_config import setup_logging, get_logger, disable_duplicate_handlers, prevent_logger_propagation_issues

# Clean up any existing duplicate handlers and initialize logging only once
disable_duplicate_handlers()
setup_logging(level=logging.INFO)
prevent_logger_propagation_issues()
logger = get_logger(__name__)


async def initialize_classification_system(
    csv_file_path: str = None, 
    config_path: str = None,
    strict_mode: bool = True
) -> 'ClassificationPipeline':
    """
    Initialize the complete classification system using the new configuration management.
    
    Args:
        csv_file_path: Path to the category CSV file
        config_path: Path to configuration file (optional)
        strict_mode: Enable strict category matching (default: True)
        
    Returns:
        Fully initialized ClassificationPipeline
    """
    from src.config.config_manager import ConfigurationManager
    from src.config.llm_factory import LLMFactory
    from src.agents.classification_pipeline import ClassificationPipeline
    from src.models.entities import ClassificationHierarchy
    from src.utils.classification_validator import ClassificationValidator
    from qdrant_client import QdrantClient
    
    logger.info("🚀 Starting ITSM Classification System initialization...")
    logger.info(f"Strict mode: {'ENABLED' if strict_mode else 'DISABLED'}")
    
    try:
        # 1. Initialize Configuration Manager
        logger.info("📋 Initializing configuration manager...")
        config_manager = ConfigurationManager(config_path)
        
        # Override CSV path if provided
        if csv_file_path:
            config_manager.raw_config["data"]["csv_file_path"] = csv_file_path
        
        # Validate configuration
        validation_result = config_manager.validate_configuration()
        if not validation_result["is_valid"]:
            logger.error("❌ Configuration validation failed:")
            for error in validation_result["errors"]:
                logger.error(f"  - {error}")
            raise ValueError("System configuration is invalid")
        
        if validation_result["warnings"]:
            logger.warning("⚠️ Configuration warnings:")
            for warning in validation_result["warnings"]:
                logger.warning(f"  - {warning}")
        
        logger.info("✅ Configuration validation successful")
        
        # 2. Test LLM Factory
        logger.info("🧪 Testing LLM factory...")
        test_config = config_manager.get_agent_config("category_classifier")
        try:
            # Test LLM creation without actually creating (just validation)
            llm_validation = LLMFactory.validate_config(test_config)
            if not llm_validation["is_valid"]:
                logger.error("❌ LLM configuration validation failed:")
                for error in llm_validation["errors"]:
                    logger.error(f"  - {error}")
                raise ValueError("LLM configuration is invalid")
            logger.info("✅ LLM factory validation successful")
        except Exception as e:
            logger.error(f"❌ LLM factory test failed: {e}")
            raise
        
        # 3. Initialize Qdrant client (optional)
        logger.info("🔗 Initializing Qdrant vector database...")
        qdrant_config = config_manager.get_qdrant_config()
        try:
            qdrant_client = QdrantClient(url=qdrant_config["url"])
            # Test connection
            collections = qdrant_client.get_collections()
            logger.info(f"✅ Qdrant connected successfully. Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant connection failed: {e}")
            logger.info("Continuing without Qdrant (will use keyword fallback)")
            qdrant_client = None
        
        # 4. Load hierarchy from CSV
        logger.info("📊 Loading classification hierarchy from CSV...")
        data_config = config_manager.get_data_config()
        csv_path = data_config["csv_file_path"]
        
        if not os.path.exists(csv_path):
            logger.error(f"❌ CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            hierarchy = ClassificationHierarchy.load_from_file(csv_path)
            
            # Get statistics about loaded hierarchy
            stats = hierarchy.get_statistics()
            logger.info(f"✅ Hierarchy loaded: {stats['total_categories']} categories, "
                       f"{stats['total_subcategories']} subcategories")
            
            # Initialize classification validator for strict mode
            if strict_mode:
                classification_validator = ClassificationValidator(hierarchy)
                logger.info("✅ Strict classification validator initialized")
                
                # Log valid categories for debugging
                valid_cats = list(hierarchy.categories.keys())
                logger.info(f"Valid categories: {', '.join(valid_cats[:5])}{'...' if len(valid_cats) > 5 else ''}")
            
            if stats.get('total_categories', 0) > 0:
                logger.info(f"Valid categories: {', '.join(list(hierarchy.categories.keys())[:5])}{'...' if len(hierarchy.categories) > 5 else ''}")
            
            # Note: ClassificationHierarchy doesn't provide warnings like CategoryLoader did
            # But we can add our own validation if needed
                    
        except Exception as e:
            logger.error(f"❌ Failed to load hierarchy from CSV: {e}")
            raise
        
        # 5. Create classification pipeline using the new configuration system
        logger.info("🤖 Initializing classification pipeline...")
        try:
            # Create pipeline with configuration manager
            pipeline = await ClassificationPipeline.create(
                config_path=None,  # Don't load config again, we already have it
                hierarchy=hierarchy,
                qdrant_client=qdrant_client
            )
            
            # Override the pipeline's configuration with our validated configuration manager
            pipeline.config_manager = config_manager
            pipeline.config = config_manager.raw_config  # For backward compatibility
            
            # Set strict mode on all classifiers
            if hasattr(pipeline, 'category_classifier') and pipeline.category_classifier:
                pipeline.category_classifier.strict_mode = strict_mode
                if strict_mode:
                    pipeline.category_classifier.classification_validator = classification_validator
                    logger.info("🎯 Adding category training examples...")
                    await pipeline.category_classifier.add_category_examples()
                
            if hasattr(pipeline, 'subcategory_classifier'):
                pipeline.subcategory_classifier.strict_mode = strict_mode
                if strict_mode:
                    pipeline.subcategory_classifier.classification_validator = classification_validator
            
            # Configure Arabic processor to remove system tags
            classification_config = config_manager.get_classification_config()
            if hasattr(pipeline, 'arabic_processor'):
                pipeline.arabic_processor.remove_system_tags = classification_config.get("remove_system_tags", True)
                pipeline.arabic_processor.system_tags_patterns = [
                    r'\(AutoClosed\)',
                    r'\(auto[_\-]?closed\)',
                    r'\[.*?closed.*?\]',
                    r'تم الإغلاق تلقائي[اً]?',
                    r'مغلق تلقائي[اً]?',
                ]
            
            logger.info("✅ Classification pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            raise
        
        # 6. Perform system health check
        logger.info("🏥 Performing system health check...")
        
        # Check all agents are properly initialized
        agents = ['orchestrator', 'arabic_processor', 'subcategory_classifier']
        for agent_name in agents:
            agent = getattr(pipeline, agent_name, None)
            if agent and hasattr(agent, 'config'):
                logger.info(f"  ✅ {agent_name} agent: initialized")
            else:
                logger.warning(f"  ⚠️ {agent_name} agent: not properly initialized")
        
        # Check category classifier (async-initialized)
        if pipeline.category_classifier:
            logger.info("  ✅ category_classifier agent: initialized")
        else:
            logger.warning("  ⚠️ category_classifier agent: not yet initialized")
        
        logger.info("🎉 ITSM Classification System initialization completed successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"💥 System initialization failed: {e}")
        raise


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
        logger.info(f"🎫 Classifying ticket: {ticket_id or 'auto-generated'}")
        
        # Use the pipeline's classify_ticket method directly
        result = await pipeline.classify_ticket(ticket_text, ticket_id)
        
        logger.info(f"✅ Classification completed for ticket {ticket_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Ticket classification failed: {e}")
        return {
            "ticket_id": ticket_id,
            "error": str(e),
            "status": "failed"
        }


def create_sample_config_file(file_path: str = "itsm_config.json"):
    """Create a sample configuration file using the configuration manager"""
    from src.config.config_manager import create_sample_config_file as create_config
    create_config(file_path)


async def main():
    """Main entry point with command line argument support"""
    parser = argparse.ArgumentParser(description="ITSM Classification System (Updated)")
    parser.add_argument("--csv", default="Category + SubCategory.csv", 
                       help="Path to category CSV file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", 
                       help="Create sample configuration file")
    parser.add_argument("--test-ticket", help="Test classification with this text")
    parser.add_argument("--strict-mode", action="store_true", default=True,
                       help="Enable strict category matching (default: True)")
    parser.add_argument("--no-strict-mode", action="store_false", dest="strict_mode",
                       help="Disable strict category matching")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config_file()
        return
    
    try:
        # Initialize system
        logger.info("=" * 60)
        logger.info("🚀 ITSM Classification System (Updated with Config Manager)")
        logger.info("=" * 60)
        
        pipeline = await initialize_classification_system(
            csv_file_path=args.csv,
            config_path=args.config,
            strict_mode=args.strict_mode
        )
        
        # Test with sample ticket if provided
        if args.test_ticket:
            logger.info("🧪 Testing classification...")
            result = await classify_ticket(pipeline, args.test_ticket)
            
            print("\n" + "=" * 50)
            print("📊 CLASSIFICATION RESULT")
            print("=" * 50)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n" + "=" * 50)
            print("✅ System initialized successfully!")
            print("💡 Use --test-ticket 'your text here' to test classification.")
            print("🔧 Use --create-config to create a sample configuration file.")
            print("=" * 50)
            
    except Exception as e:
        logger.error(f"💥 System initialization failed: {e}")
        print(f"\n❌ Error: {e}")
        print("🔧 Try using --create-config to create a proper configuration file.")
        exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
