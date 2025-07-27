"""import asyncio
import logging
from main import initialize_classification_system, classify_ticket

# Use centralized logging (main.py already sets it up)
from src.utils.logging_config import get_logger
logger = get_logger(__name__)stic test to check if strict validation is working
"""

import asyncio
import logging
from main import initialize_classification_system

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def diagnostic_test():
    """Quick diagnostic test"""
    
    logger.info("=== DIAGNOSTIC TEST ===")
    
    # Initialize system
    pipeline = await initialize_classification_system(strict_mode=True)
    
    # Check if strict mode is actually set
    if hasattr(pipeline, 'category_classifier'):
        strict_mode = getattr(pipeline.category_classifier, 'strict_mode', 'NOT_SET')
        validator = getattr(pipeline.category_classifier, 'classification_validator', 'NOT_SET')
        logger.info(f"Category classifier strict_mode: {strict_mode}")
        logger.info(f"Category classifier validator: {validator}")
        
        # Get valid categories
        if hasattr(pipeline.category_classifier, 'hierarchy'):
            valid_cats = list(pipeline.category_classifier.hierarchy.categories.keys())
            logger.info(f"Valid categories ({len(valid_cats)}): {valid_cats[:5]}...")
            
            # Check if expected categories exist
            test_categories = ["التسجيل", "الإرسالية", "تسجيل الدخول"]
            for cat in test_categories:
                exists = cat in valid_cats
                logger.info(f"  '{cat}': {'✅ EXISTS' if exists else '❌ MISSING'}")
    
    # Test system tag removal
    if hasattr(pipeline, 'arabic_processor'):
        test_text = "مشكلة في النظام (AutoClosed)"
        if hasattr(pipeline.arabic_processor, '_remove_system_tags'):
            cleaned = pipeline.arabic_processor._remove_system_tags(test_text)
            logger.info(f"Tag removal test:")
            logger.info(f"  Original: {test_text}")
            logger.info(f"  Cleaned:  {cleaned}")
            logger.info(f"  Success: {'✅' if '(AutoClosed)' not in cleaned else '❌'}")
        else:
            logger.info("❌ _remove_system_tags method not found")

if __name__ == "__main__":
    asyncio.run(diagnostic_test())
