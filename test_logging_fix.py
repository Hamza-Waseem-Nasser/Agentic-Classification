"""
Simple test to verify logging fix - no duplicate messages
"""
import asyncio
from main import initialize_classification_system, classify_ticket
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

async def test_logging_fix():
    """Test that logging doesn't show duplicate messages"""
    
    logger.info("=" * 50)
    logger.info("TESTING LOGGING FIX")
    logger.info("=" * 50)
    logger.info("This message should appear only ONCE")
    
    # Initialize system (this used to cause duplicate logging)
    logger.info("Initializing classification system...")
    pipeline = await initialize_classification_system(strict_mode=True)
    logger.info("System initialized successfully")
    
    # Test classification (this also used to cause duplicates)
    test_text = "مشكلة في تسجيل الدخول"
    logger.info(f"Testing classification with: {test_text}")
    
    result = await classify_ticket(pipeline, test_text, "test_001")
    
    category = result.get('classification', {}).get('category', 'Unknown')
    logger.info(f"Classification result: {category}")
    logger.info("=" * 50)
    logger.info("TEST COMPLETE - Check above for duplicate messages")
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_logging_fix())
